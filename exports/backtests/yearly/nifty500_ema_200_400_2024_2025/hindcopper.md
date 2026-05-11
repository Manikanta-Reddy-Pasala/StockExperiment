# Hindustan Copper Ltd. (HINDCOPPER)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 568.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT2_SKIP | 6 |
| ALERT3 | 54 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 42 |
| PARTIAL | 18 |
| TARGET_HIT | 7 |
| STOP_HIT | 36 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 61 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 36 / 25
- **Target hits / Stop hits / Partials:** 7 / 36 / 18
- **Avg / median % per leg:** 1.78% / 1.56%
- **Sum % (uncompounded):** 108.85%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.51% | -18.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.51% | -18.1% |
| SELL (all) | 57 | 36 | 63.2% | 7 | 32 | 18 | 2.23% | 126.9% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| SELL @ 3rd Alert (retest2) | 55 | 34 | 61.8% | 6 | 32 | 17 | 2.03% | 111.9% |
| retest1 (combined) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| retest2 (combined) | 59 | 34 | 57.6% | 6 | 36 | 17 | 1.59% | 93.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-07-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 15:15:00 | 327.15 | 335.95 | 335.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-05 09:15:00 | 325.00 | 335.84 | 335.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-08 10:15:00 | 335.50 | 335.50 | 335.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-08 10:15:00 | 335.50 | 335.50 | 335.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 335.50 | 335.50 | 335.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-08 10:45:00 | 337.70 | 335.50 | 335.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 11:15:00 | 336.70 | 335.51 | 335.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-08 11:45:00 | 337.50 | 335.51 | 335.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 12:15:00 | 338.90 | 335.54 | 335.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-08 12:30:00 | 338.10 | 335.54 | 335.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2024-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 11:15:00 | 347.15 | 336.01 | 335.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 12:15:00 | 348.75 | 336.13 | 336.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 09:15:00 | 332.60 | 336.44 | 336.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 09:15:00 | 332.60 | 336.44 | 336.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 332.60 | 336.44 | 336.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:00:00 | 332.60 | 336.44 | 336.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 332.45 | 336.40 | 336.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 327.80 | 336.40 | 336.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 336.20 | 336.24 | 336.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 10:30:00 | 335.40 | 336.24 | 336.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 11:15:00 | 336.05 | 336.24 | 336.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 12:00:00 | 336.05 | 336.24 | 336.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 12:15:00 | 335.65 | 336.24 | 336.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 12:30:00 | 334.40 | 336.24 | 336.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 13:15:00 | 334.80 | 336.22 | 336.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 14:00:00 | 334.80 | 336.22 | 336.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2024-07-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 13:15:00 | 331.60 | 335.98 | 335.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-12 14:15:00 | 330.35 | 335.93 | 335.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-01 09:15:00 | 327.15 | 325.65 | 329.82 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-01 10:30:00 | 324.30 | 325.64 | 329.79 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 308.08 | 324.16 | 328.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2024-08-05 11:15:00 | 291.87 | 323.55 | 328.42 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 4 — BUY (started 2024-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 11:15:00 | 340.85 | 321.63 | 321.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-26 14:15:00 | 343.85 | 322.23 | 321.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-07 09:15:00 | 320.30 | 328.01 | 325.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-07 09:15:00 | 320.30 | 328.01 | 325.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 320.30 | 328.01 | 325.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 09:45:00 | 322.25 | 328.01 | 325.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 10:15:00 | 315.70 | 327.89 | 325.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 11:00:00 | 315.70 | 327.89 | 325.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 321.65 | 323.55 | 323.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 10:00:00 | 321.65 | 323.55 | 323.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 321.75 | 323.53 | 323.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 10:30:00 | 322.30 | 323.53 | 323.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 11:15:00 | 321.30 | 323.51 | 323.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 12:15:00 | 320.25 | 323.51 | 323.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 10:15:00 | 320.30 | 323.28 | 323.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 11:00:00 | 320.30 | 323.28 | 323.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 15:15:00 | 320.50 | 323.13 | 323.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 09:30:00 | 320.00 | 323.09 | 323.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2024-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 12:15:00 | 317.85 | 322.94 | 322.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 15:15:00 | 314.00 | 322.76 | 322.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 14:15:00 | 324.25 | 322.57 | 322.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 14:15:00 | 324.25 | 322.57 | 322.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 324.25 | 322.57 | 322.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 14:45:00 | 325.75 | 322.57 | 322.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 15:15:00 | 321.00 | 322.55 | 322.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:15:00 | 321.05 | 322.55 | 322.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 318.35 | 322.51 | 322.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 12:00:00 | 316.05 | 322.40 | 322.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 13:30:00 | 315.95 | 322.27 | 322.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 14:00:00 | 315.80 | 322.27 | 322.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 14:30:00 | 316.25 | 322.21 | 322.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 14:15:00 | 300.25 | 321.17 | 322.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 14:15:00 | 300.15 | 321.17 | 322.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 14:15:00 | 300.01 | 321.17 | 322.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 14:15:00 | 300.44 | 321.17 | 322.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-25 09:15:00 | 284.44 | 316.78 | 319.69 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 6 — BUY (started 2025-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 14:15:00 | 245.65 | 224.23 | 224.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 253.69 | 224.72 | 224.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 243.69 | 245.41 | 237.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 12:45:00 | 243.95 | 245.41 | 237.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 15:15:00 | 256.20 | 265.47 | 256.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:15:00 | 256.95 | 265.47 | 256.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 256.50 | 265.38 | 256.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 13:00:00 | 259.10 | 264.22 | 256.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 11:30:00 | 259.25 | 263.86 | 256.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 15:00:00 | 261.90 | 263.69 | 256.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 09:15:00 | 248.60 | 263.52 | 256.40 | SL hit (close<static) qty=1.00 sl=254.55 alert=retest2 |

### Cycle 7 — SELL (started 2025-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 13:15:00 | 239.80 | 251.54 | 251.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 15:15:00 | 239.11 | 248.79 | 250.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 09:15:00 | 243.80 | 242.87 | 246.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 09:15:00 | 243.80 | 242.87 | 246.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 243.80 | 242.87 | 246.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:45:00 | 243.73 | 242.87 | 246.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 12:15:00 | 247.00 | 242.96 | 246.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:00:00 | 247.00 | 242.96 | 246.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 248.98 | 243.02 | 246.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:45:00 | 249.45 | 243.02 | 246.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 247.51 | 243.21 | 246.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:00:00 | 247.51 | 243.21 | 246.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 245.38 | 243.26 | 246.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:30:00 | 245.82 | 243.26 | 246.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 247.54 | 243.26 | 246.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:30:00 | 248.26 | 243.26 | 246.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 246.33 | 243.30 | 246.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 12:00:00 | 245.01 | 243.31 | 246.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 248.20 | 243.43 | 246.20 | SL hit (close>static) qty=1.00 sl=247.68 alert=retest2 |

### Cycle 8 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 283.10 | 248.59 | 248.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 11:15:00 | 289.76 | 249.37 | 248.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 325.45 | 329.01 | 307.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-06 10:00:00 | 325.45 | 329.01 | 307.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 314.25 | 331.14 | 316.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 15:00:00 | 314.25 | 331.14 | 316.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 15:15:00 | 313.60 | 330.96 | 316.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:15:00 | 318.55 | 330.96 | 316.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 313.80 | 330.64 | 316.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:00:00 | 313.80 | 330.64 | 316.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 521.75 | 564.47 | 535.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 10:00:00 | 521.75 | 564.47 | 535.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 527.00 | 564.10 | 535.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 14:45:00 | 528.75 | 562.59 | 535.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 503.30 | 561.66 | 534.96 | SL hit (close<static) qty=1.00 sl=519.40 alert=retest2 |

### Cycle 9 — SELL (started 2026-03-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 13:15:00 | 458.75 | 516.63 | 516.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 455.00 | 516.01 | 516.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 526.75 | 511.34 | 513.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 526.75 | 511.34 | 513.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 526.75 | 511.34 | 513.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 13:15:00 | 520.70 | 512.81 | 514.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 14:15:00 | 520.75 | 512.90 | 514.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 14:45:00 | 520.80 | 513.00 | 514.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 533.80 | 513.31 | 514.68 | SL hit (close>static) qty=1.00 sl=533.60 alert=retest2 |

### Cycle 10 — BUY (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 10:15:00 | 554.40 | 516.04 | 515.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 10:15:00 | 561.40 | 518.62 | 517.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 14:15:00 | 534.40 | 537.37 | 528.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 15:00:00 | 534.40 | 537.37 | 528.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 13:15:00 | 535.80 | 537.42 | 529.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 13:30:00 | 532.85 | 537.42 | 529.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-08-01 10:30:00 | 324.30 | 2024-08-05 09:15:00 | 308.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-08-01 10:30:00 | 324.30 | 2024-08-05 11:15:00 | 291.87 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-08-19 13:15:00 | 319.10 | 2024-08-19 13:15:00 | 323.00 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2024-08-20 10:45:00 | 319.45 | 2024-08-26 09:15:00 | 323.50 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2024-08-20 11:30:00 | 319.10 | 2024-08-26 09:15:00 | 323.50 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2024-08-21 12:45:00 | 318.05 | 2024-08-26 09:15:00 | 323.50 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2024-08-29 13:45:00 | 321.50 | 2024-08-29 14:15:00 | 330.00 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2024-09-02 14:45:00 | 320.40 | 2024-09-09 09:15:00 | 304.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-03 09:30:00 | 320.80 | 2024-09-09 09:15:00 | 304.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-03 11:30:00 | 320.70 | 2024-09-09 09:15:00 | 304.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-02 14:45:00 | 320.40 | 2024-09-13 12:15:00 | 317.40 | STOP_HIT | 0.50 | 0.94% |
| SELL | retest2 | 2024-09-03 09:30:00 | 320.80 | 2024-09-13 12:15:00 | 317.40 | STOP_HIT | 0.50 | 1.06% |
| SELL | retest2 | 2024-09-03 11:30:00 | 320.70 | 2024-09-13 12:15:00 | 317.40 | STOP_HIT | 0.50 | 1.03% |
| SELL | retest2 | 2024-09-13 11:30:00 | 314.25 | 2024-09-16 12:15:00 | 321.05 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2024-09-13 15:15:00 | 314.50 | 2024-09-16 12:15:00 | 321.05 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2024-09-16 11:30:00 | 314.20 | 2024-09-16 12:15:00 | 321.05 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2024-09-19 11:00:00 | 314.55 | 2024-09-19 14:15:00 | 323.20 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2024-10-21 12:00:00 | 316.05 | 2024-10-22 14:15:00 | 300.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 13:30:00 | 315.95 | 2024-10-22 14:15:00 | 300.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 14:00:00 | 315.80 | 2024-10-22 14:15:00 | 300.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 14:30:00 | 316.25 | 2024-10-22 14:15:00 | 300.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 12:00:00 | 316.05 | 2024-10-25 09:15:00 | 284.44 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-21 13:30:00 | 315.95 | 2024-10-25 09:15:00 | 284.36 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-21 14:00:00 | 315.80 | 2024-10-25 09:15:00 | 284.22 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-21 14:30:00 | 316.25 | 2024-10-25 09:15:00 | 284.62 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-10 12:15:00 | 290.00 | 2024-12-18 12:15:00 | 275.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-11 13:30:00 | 290.20 | 2024-12-18 12:15:00 | 275.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-11 15:15:00 | 289.05 | 2024-12-18 12:15:00 | 274.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-12 15:15:00 | 289.55 | 2024-12-18 12:15:00 | 275.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-13 09:15:00 | 285.15 | 2024-12-19 09:15:00 | 274.26 | PARTIAL | 0.50 | 3.82% |
| SELL | retest2 | 2024-12-16 09:30:00 | 288.70 | 2024-12-19 09:15:00 | 274.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 10:45:00 | 288.70 | 2024-12-19 09:15:00 | 274.31 | PARTIAL | 0.50 | 4.98% |
| SELL | retest2 | 2024-12-10 12:15:00 | 290.00 | 2024-12-20 15:15:00 | 261.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-11 13:30:00 | 290.20 | 2024-12-20 15:15:00 | 261.18 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-16 15:15:00 | 288.75 | 2024-12-20 15:15:00 | 270.89 | PARTIAL | 0.50 | 6.18% |
| SELL | retest2 | 2024-12-11 15:15:00 | 289.05 | 2024-12-24 11:15:00 | 284.20 | STOP_HIT | 0.50 | 1.68% |
| SELL | retest2 | 2024-12-12 15:15:00 | 289.55 | 2024-12-24 11:15:00 | 284.20 | STOP_HIT | 0.50 | 1.85% |
| SELL | retest2 | 2024-12-13 09:15:00 | 285.15 | 2024-12-24 11:15:00 | 284.20 | STOP_HIT | 0.50 | 0.33% |
| SELL | retest2 | 2024-12-16 09:30:00 | 288.70 | 2024-12-24 11:15:00 | 284.20 | STOP_HIT | 0.50 | 1.56% |
| SELL | retest2 | 2024-12-16 10:45:00 | 288.70 | 2024-12-24 11:15:00 | 284.20 | STOP_HIT | 0.50 | 1.56% |
| SELL | retest2 | 2024-12-16 15:15:00 | 288.75 | 2024-12-24 11:15:00 | 284.20 | STOP_HIT | 0.50 | 1.58% |
| SELL | retest2 | 2025-03-20 14:30:00 | 232.80 | 2025-03-21 09:15:00 | 237.00 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-03-20 15:00:00 | 231.75 | 2025-03-21 09:15:00 | 237.00 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2025-03-21 11:15:00 | 231.40 | 2025-03-24 13:15:00 | 238.18 | STOP_HIT | 1.00 | -2.93% |
| SELL | retest2 | 2025-03-24 09:30:00 | 232.75 | 2025-03-24 13:15:00 | 238.18 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-03-24 11:15:00 | 232.60 | 2025-03-24 13:15:00 | 238.18 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2025-03-25 10:00:00 | 231.82 | 2025-03-27 14:15:00 | 220.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-26 11:00:00 | 232.50 | 2025-03-27 14:15:00 | 220.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-25 10:00:00 | 231.82 | 2025-03-27 15:15:00 | 224.80 | STOP_HIT | 0.50 | 3.03% |
| SELL | retest2 | 2025-03-26 11:00:00 | 232.50 | 2025-03-27 15:15:00 | 224.80 | STOP_HIT | 0.50 | 3.31% |
| SELL | retest2 | 2025-05-15 09:30:00 | 230.75 | 2025-05-23 12:15:00 | 236.81 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2025-07-29 13:00:00 | 259.10 | 2025-07-31 09:15:00 | 248.60 | STOP_HIT | 1.00 | -4.05% |
| BUY | retest2 | 2025-07-30 11:30:00 | 259.25 | 2025-07-31 09:15:00 | 248.60 | STOP_HIT | 1.00 | -4.11% |
| BUY | retest2 | 2025-07-30 15:00:00 | 261.90 | 2025-07-31 09:15:00 | 248.60 | STOP_HIT | 1.00 | -5.08% |
| SELL | retest2 | 2025-09-08 12:00:00 | 245.01 | 2025-09-10 09:15:00 | 248.20 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-09-10 15:15:00 | 244.00 | 2025-09-11 10:15:00 | 251.55 | STOP_HIT | 1.00 | -3.09% |
| BUY | retest2 | 2026-03-12 14:45:00 | 528.75 | 2026-03-13 09:15:00 | 503.30 | STOP_HIT | 1.00 | -4.81% |
| SELL | retest2 | 2026-04-09 13:15:00 | 520.70 | 2026-04-10 09:15:00 | 533.80 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2026-04-09 14:15:00 | 520.75 | 2026-04-10 09:15:00 | 533.80 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2026-04-09 14:45:00 | 520.80 | 2026-04-10 09:15:00 | 533.80 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2026-04-13 09:15:00 | 519.85 | 2026-04-15 09:15:00 | 557.95 | STOP_HIT | 1.00 | -7.33% |
