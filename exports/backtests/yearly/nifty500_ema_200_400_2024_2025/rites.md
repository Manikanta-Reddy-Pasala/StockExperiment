# RITES Ltd. (RITES)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 226.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT2_SKIP | 3 |
| ALERT3 | 28 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 25 |
| PARTIAL | 20 |
| TARGET_HIT | 17 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 45 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 36 / 9
- **Target hits / Stop hits / Partials:** 16 / 9 / 20
- **Avg / median % per leg:** 4.89% / 5.18%
- **Sum % (uncompounded):** 220.13%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 45 | 36 | 80.0% | 16 | 9 | 20 | 4.89% | 220.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 45 | 36 | 80.0% | 16 | 9 | 20 | 4.89% | 220.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 45 | 36 | 80.0% | 16 | 9 | 20 | 4.89% | 220.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-16 12:15:00 | 330.48 | 350.86 | 350.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-16 13:15:00 | 328.48 | 350.64 | 350.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 09:15:00 | 339.15 | 336.60 | 342.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-05 10:00:00 | 339.15 | 336.60 | 342.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 339.20 | 336.12 | 341.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 09:15:00 | 337.65 | 336.28 | 341.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-13 09:15:00 | 345.48 | 336.90 | 341.28 | SL hit (close>static) qty=1.00 sl=342.00 alert=retest2 |

### Cycle 2 — BUY (started 2024-09-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 12:15:00 | 359.95 | 344.55 | 344.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 13:15:00 | 361.15 | 344.71 | 344.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 09:15:00 | 345.85 | 347.62 | 346.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 09:15:00 | 345.85 | 347.62 | 346.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 345.85 | 347.62 | 346.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 09:45:00 | 344.70 | 347.62 | 346.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 346.65 | 347.61 | 346.15 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2024-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 10:15:00 | 321.30 | 344.81 | 344.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 12:15:00 | 317.55 | 344.31 | 344.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 301.40 | 291.99 | 307.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 09:15:00 | 301.40 | 291.99 | 307.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 301.40 | 291.99 | 307.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-26 11:15:00 | 286.35 | 292.02 | 306.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-26 14:00:00 | 286.15 | 291.84 | 306.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-26 14:45:00 | 286.50 | 291.79 | 306.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 11:45:00 | 285.80 | 291.70 | 304.82 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 304.35 | 290.71 | 301.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 14:15:00 | 301.75 | 291.21 | 301.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 09:15:00 | 298.80 | 291.45 | 301.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 09:30:00 | 301.40 | 292.82 | 301.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 15:15:00 | 301.65 | 293.90 | 301.60 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 299.20 | 294.03 | 301.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 14:00:00 | 297.25 | 294.22 | 301.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 10:00:00 | 297.25 | 294.36 | 301.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 11:30:00 | 297.80 | 294.42 | 301.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 13:15:00 | 286.66 | 294.03 | 300.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 13:15:00 | 283.86 | 294.03 | 300.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 13:15:00 | 286.33 | 294.03 | 300.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 13:15:00 | 286.57 | 294.03 | 300.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 14:15:00 | 282.39 | 293.89 | 300.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 14:15:00 | 282.39 | 293.89 | 300.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 14:15:00 | 282.91 | 293.89 | 300.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 13:15:00 | 272.03 | 289.60 | 297.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 13:15:00 | 271.84 | 289.60 | 297.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 13:15:00 | 272.18 | 289.60 | 297.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 13:15:00 | 271.51 | 289.60 | 297.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-12-30 13:15:00 | 271.57 | 289.60 | 297.24 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2025-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 11:15:00 | 296.34 | 234.98 | 234.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 09:15:00 | 305.35 | 262.47 | 251.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 10:15:00 | 276.45 | 278.14 | 264.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 11:00:00 | 276.45 | 278.14 | 264.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 273.15 | 278.93 | 272.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:45:00 | 273.25 | 278.93 | 272.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 273.00 | 278.87 | 272.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 11:00:00 | 273.00 | 278.87 | 272.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 11:15:00 | 272.70 | 278.81 | 272.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 11:30:00 | 272.60 | 278.81 | 272.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 12:15:00 | 273.00 | 278.75 | 272.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 12:30:00 | 273.15 | 278.75 | 272.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 272.60 | 278.69 | 272.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 13:30:00 | 272.85 | 278.69 | 272.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 273.30 | 278.64 | 272.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 15:00:00 | 273.30 | 278.64 | 272.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 273.15 | 278.45 | 272.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 12:00:00 | 273.15 | 278.45 | 272.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 272.95 | 278.40 | 272.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:00:00 | 272.95 | 278.40 | 272.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 273.65 | 278.35 | 272.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:30:00 | 273.20 | 278.35 | 272.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 270.70 | 278.18 | 272.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:45:00 | 270.90 | 278.18 | 272.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 269.80 | 278.10 | 272.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:30:00 | 269.60 | 278.10 | 272.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 10:15:00 | 250.40 | 268.93 | 269.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 248.70 | 268.55 | 268.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 14:15:00 | 258.29 | 257.39 | 261.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 15:00:00 | 258.29 | 257.39 | 261.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 263.08 | 257.49 | 261.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 12:15:00 | 258.55 | 258.30 | 261.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 13:00:00 | 258.23 | 258.30 | 261.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 14:15:00 | 269.55 | 259.22 | 261.88 | SL hit (close>static) qty=1.00 sl=267.99 alert=retest2 |

### Cycle 6 — BUY (started 2025-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 15:15:00 | 270.11 | 263.92 | 263.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 11:15:00 | 271.50 | 264.11 | 264.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 09:15:00 | 263.10 | 264.56 | 264.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 09:15:00 | 263.10 | 264.56 | 264.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 263.10 | 264.56 | 264.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:45:00 | 263.10 | 264.56 | 264.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 262.90 | 264.54 | 264.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:00:00 | 262.90 | 264.54 | 264.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2025-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 10:15:00 | 255.91 | 263.88 | 263.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 11:15:00 | 254.76 | 263.79 | 263.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 11:15:00 | 249.71 | 249.39 | 253.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-12 12:00:00 | 249.71 | 249.39 | 253.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 252.83 | 248.99 | 253.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:45:00 | 253.67 | 248.99 | 253.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 252.28 | 249.02 | 253.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:30:00 | 253.06 | 249.02 | 253.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 252.50 | 249.23 | 252.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 14:30:00 | 247.54 | 249.24 | 252.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 15:00:00 | 247.29 | 249.24 | 252.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-25 12:15:00 | 235.16 | 248.52 | 251.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-25 12:15:00 | 234.93 | 248.52 | 251.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-08 11:15:00 | 222.79 | 241.02 | 246.84 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 8 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 226.18 | 212.76 | 212.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 12:15:00 | 229.15 | 214.09 | 213.43 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-09-11 09:15:00 | 337.65 | 2024-09-13 09:15:00 | 345.48 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2024-09-19 14:30:00 | 337.43 | 2024-09-20 09:15:00 | 363.45 | STOP_HIT | 1.00 | -7.71% |
| SELL | retest2 | 2024-11-26 11:15:00 | 286.35 | 2024-12-20 13:15:00 | 286.66 | PARTIAL | 0.50 | -0.11% |
| SELL | retest2 | 2024-11-26 14:00:00 | 286.15 | 2024-12-20 13:15:00 | 283.86 | PARTIAL | 0.50 | 0.80% |
| SELL | retest2 | 2024-11-26 14:45:00 | 286.50 | 2024-12-20 13:15:00 | 286.33 | PARTIAL | 0.50 | 0.06% |
| SELL | retest2 | 2024-11-29 11:45:00 | 285.80 | 2024-12-20 13:15:00 | 286.57 | PARTIAL | 0.50 | -0.27% |
| SELL | retest2 | 2024-12-09 14:15:00 | 301.75 | 2024-12-20 14:15:00 | 282.39 | PARTIAL | 0.50 | 6.42% |
| SELL | retest2 | 2024-12-10 09:15:00 | 298.80 | 2024-12-20 14:15:00 | 282.39 | PARTIAL | 0.50 | 5.49% |
| SELL | retest2 | 2024-12-12 09:30:00 | 301.40 | 2024-12-20 14:15:00 | 282.91 | PARTIAL | 0.50 | 6.13% |
| SELL | retest2 | 2024-12-16 15:15:00 | 301.65 | 2024-12-30 13:15:00 | 272.03 | PARTIAL | 0.50 | 9.82% |
| SELL | retest2 | 2024-12-17 14:00:00 | 297.25 | 2024-12-30 13:15:00 | 271.84 | PARTIAL | 0.50 | 8.55% |
| SELL | retest2 | 2024-12-18 10:00:00 | 297.25 | 2024-12-30 13:15:00 | 272.18 | PARTIAL | 0.50 | 8.44% |
| SELL | retest2 | 2024-12-18 11:30:00 | 297.80 | 2024-12-30 13:15:00 | 271.51 | PARTIAL | 0.50 | 8.83% |
| SELL | retest2 | 2024-11-26 11:15:00 | 286.35 | 2024-12-30 13:15:00 | 271.57 | TARGET_HIT | 0.50 | 5.16% |
| SELL | retest2 | 2024-11-26 14:00:00 | 286.15 | 2024-12-30 13:15:00 | 268.92 | TARGET_HIT | 0.50 | 6.02% |
| SELL | retest2 | 2024-11-26 14:45:00 | 286.50 | 2024-12-30 13:15:00 | 271.26 | TARGET_HIT | 0.50 | 5.32% |
| SELL | retest2 | 2024-11-29 11:45:00 | 285.80 | 2024-12-30 13:15:00 | 271.49 | TARGET_HIT | 0.50 | 5.01% |
| SELL | retest2 | 2024-12-09 14:15:00 | 301.75 | 2024-12-30 14:15:00 | 267.53 | TARGET_HIT | 0.50 | 11.34% |
| SELL | retest2 | 2024-12-10 09:15:00 | 298.80 | 2024-12-30 14:15:00 | 267.53 | TARGET_HIT | 0.50 | 10.47% |
| SELL | retest2 | 2024-12-12 09:30:00 | 301.40 | 2024-12-30 14:15:00 | 268.02 | TARGET_HIT | 0.50 | 11.07% |
| SELL | retest2 | 2024-12-16 15:15:00 | 301.65 | 2024-12-31 10:15:00 | 297.65 | STOP_HIT | 0.50 | 1.33% |
| SELL | retest2 | 2024-12-17 14:00:00 | 297.25 | 2024-12-31 10:15:00 | 297.65 | STOP_HIT | 0.50 | -0.13% |
| SELL | retest2 | 2024-12-18 10:00:00 | 297.25 | 2024-12-31 10:15:00 | 297.65 | STOP_HIT | 0.50 | -0.13% |
| SELL | retest2 | 2024-12-18 11:30:00 | 297.80 | 2024-12-31 10:15:00 | 297.65 | STOP_HIT | 0.50 | 0.05% |
| SELL | retest2 | 2024-12-31 11:45:00 | 293.75 | 2025-01-08 12:15:00 | 279.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-01 11:30:00 | 290.25 | 2025-01-09 12:15:00 | 276.45 | PARTIAL | 0.50 | 4.75% |
| SELL | retest2 | 2025-01-01 13:00:00 | 291.00 | 2025-01-09 14:15:00 | 275.74 | PARTIAL | 0.50 | 5.24% |
| SELL | retest2 | 2025-01-06 09:15:00 | 287.60 | 2025-01-09 15:15:00 | 273.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-31 11:45:00 | 293.75 | 2025-01-13 09:15:00 | 264.38 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-01 11:30:00 | 290.25 | 2025-01-13 13:15:00 | 261.23 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-01 13:00:00 | 291.00 | 2025-01-13 13:15:00 | 261.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-06 09:15:00 | 287.60 | 2025-01-13 13:15:00 | 258.84 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-04 12:15:00 | 258.55 | 2025-09-09 14:15:00 | 269.55 | STOP_HIT | 1.00 | -4.25% |
| SELL | retest2 | 2025-09-04 13:00:00 | 258.23 | 2025-09-09 14:15:00 | 269.55 | STOP_HIT | 1.00 | -4.38% |
| SELL | retest2 | 2025-11-21 14:30:00 | 247.54 | 2025-11-25 12:15:00 | 235.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-21 15:00:00 | 247.29 | 2025-11-25 12:15:00 | 234.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-21 14:30:00 | 247.54 | 2025-12-08 11:15:00 | 222.79 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-21 15:00:00 | 247.29 | 2025-12-08 11:15:00 | 222.56 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-23 12:45:00 | 246.15 | 2025-12-31 09:15:00 | 247.55 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-12-29 09:45:00 | 246.61 | 2026-01-09 09:15:00 | 233.84 | PARTIAL | 0.50 | 5.18% |
| SELL | retest2 | 2025-12-30 12:15:00 | 239.32 | 2026-01-09 09:15:00 | 234.28 | PARTIAL | 0.50 | 2.11% |
| SELL | retest2 | 2026-01-08 10:00:00 | 239.80 | 2026-01-12 09:15:00 | 227.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-29 09:45:00 | 246.61 | 2026-01-20 13:15:00 | 221.53 | TARGET_HIT | 0.50 | 10.17% |
| SELL | retest2 | 2025-12-30 12:15:00 | 239.32 | 2026-01-20 13:15:00 | 221.95 | TARGET_HIT | 0.50 | 7.26% |
| SELL | retest2 | 2026-01-08 10:00:00 | 239.80 | 2026-01-27 09:15:00 | 215.82 | TARGET_HIT | 0.50 | 10.00% |
