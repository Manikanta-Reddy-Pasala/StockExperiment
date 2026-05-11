# Crompton Greaves Consumer Electricals Ltd. (CROMPTON)

## Backtest Summary

- **Window:** 2022-04-07 14:15:00 → 2026-05-08 15:15:00 (7049 bars)
- **Last close:** 293.55
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT2_SKIP | 0 |
| ALERT3 | 67 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 60 |
| PARTIAL | 11 |
| TARGET_HIT | 10 |
| STOP_HIT | 50 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 71 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 22 / 49
- **Target hits / Stop hits / Partials:** 10 / 50 / 11
- **Avg / median % per leg:** 1.04% / -1.15%
- **Sum % (uncompounded):** 73.90%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 26 | 1 | 3.8% | 0 | 26 | 0 | -1.95% | -50.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 26 | 1 | 3.8% | 0 | 26 | 0 | -1.95% | -50.6% |
| SELL (all) | 45 | 21 | 46.7% | 10 | 24 | 11 | 2.77% | 124.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 45 | 21 | 46.7% | 10 | 24 | 11 | 2.77% | 124.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 71 | 22 | 31.0% | 10 | 50 | 11 | 1.04% | 73.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-06-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-30 12:15:00 | 290.70 | 283.01 | 282.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-10 12:15:00 | 292.75 | 284.67 | 283.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-25 12:15:00 | 287.95 | 288.88 | 286.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-25 13:00:00 | 287.95 | 288.88 | 286.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 13:15:00 | 286.95 | 288.86 | 286.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-25 14:00:00 | 286.95 | 288.86 | 286.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 14:15:00 | 287.40 | 288.85 | 286.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-26 09:15:00 | 288.15 | 288.84 | 286.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-26 11:00:00 | 288.25 | 288.82 | 286.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-26 11:30:00 | 288.35 | 288.82 | 286.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-27 14:45:00 | 288.25 | 289.01 | 286.93 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 13:15:00 | 287.45 | 289.78 | 287.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 14:00:00 | 287.45 | 289.78 | 287.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 14:15:00 | 287.15 | 289.75 | 287.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 14:45:00 | 286.75 | 289.75 | 287.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 15:15:00 | 287.10 | 289.73 | 287.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-03 09:15:00 | 288.70 | 289.73 | 287.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 10:15:00 | 293.25 | 289.73 | 287.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-03 11:30:00 | 295.00 | 289.78 | 287.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-04 09:30:00 | 294.25 | 289.94 | 287.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-14 11:15:00 | 297.05 | 294.06 | 290.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-16 09:15:00 | 284.60 | 294.17 | 290.65 | SL hit (close<static) qty=1.00 sl=286.60 alert=retest2 |

### Cycle 2 — SELL (started 2023-10-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-25 12:15:00 | 280.55 | 298.13 | 298.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-01 11:15:00 | 278.65 | 293.98 | 295.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-17 10:15:00 | 288.10 | 287.95 | 291.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-17 11:15:00 | 287.80 | 287.95 | 291.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 09:15:00 | 289.25 | 287.86 | 291.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-21 09:30:00 | 291.40 | 287.86 | 291.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 11:15:00 | 291.20 | 287.92 | 291.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-21 12:00:00 | 291.20 | 287.92 | 291.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 12:15:00 | 290.20 | 287.94 | 291.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-22 12:00:00 | 289.75 | 288.12 | 291.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-23 09:30:00 | 289.70 | 288.16 | 291.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-01 15:15:00 | 289.00 | 287.38 | 290.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-04 10:15:00 | 291.65 | 287.48 | 290.32 | SL hit (close>static) qty=1.00 sl=291.40 alert=retest2 |

### Cycle 3 — BUY (started 2023-12-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-20 12:15:00 | 305.35 | 292.26 | 292.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-29 11:15:00 | 310.90 | 294.92 | 293.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-18 09:15:00 | 308.75 | 309.59 | 303.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-18 09:45:00 | 307.90 | 309.59 | 303.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 12:15:00 | 303.70 | 309.74 | 303.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 13:15:00 | 302.75 | 309.74 | 303.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 13:15:00 | 301.40 | 309.65 | 303.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 14:00:00 | 301.40 | 309.65 | 303.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 14:15:00 | 300.20 | 309.56 | 303.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 14:30:00 | 300.15 | 309.56 | 303.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 10:15:00 | 305.65 | 309.37 | 303.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-24 11:45:00 | 306.30 | 309.34 | 303.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-25 10:15:00 | 302.40 | 309.17 | 303.95 | SL hit (close<static) qty=1.00 sl=302.45 alert=retest2 |

### Cycle 4 — SELL (started 2024-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 14:15:00 | 283.15 | 300.97 | 301.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-14 10:15:00 | 282.55 | 299.37 | 300.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-23 13:15:00 | 295.65 | 295.09 | 297.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-23 13:45:00 | 294.70 | 295.09 | 297.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 13:15:00 | 296.55 | 294.02 | 296.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 14:00:00 | 296.55 | 294.02 | 296.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 14:15:00 | 295.30 | 294.03 | 296.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 14:30:00 | 296.95 | 294.03 | 296.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 15:15:00 | 295.70 | 294.04 | 296.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-02 09:15:00 | 297.70 | 294.04 | 296.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 09:15:00 | 300.50 | 294.11 | 296.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-02 09:30:00 | 301.05 | 294.11 | 296.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 11:15:00 | 300.75 | 294.18 | 296.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-02 11:30:00 | 301.85 | 294.18 | 296.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 11:15:00 | 296.10 | 294.67 | 296.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-05 12:15:00 | 296.00 | 294.67 | 296.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-05 13:00:00 | 296.00 | 294.68 | 296.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-05 13:30:00 | 295.25 | 294.68 | 296.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-07 11:45:00 | 295.50 | 294.40 | 296.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 11:15:00 | 281.20 | 293.33 | 295.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 11:15:00 | 281.20 | 293.33 | 295.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 13:15:00 | 280.49 | 293.07 | 295.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 13:15:00 | 280.72 | 293.07 | 295.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-03-19 14:15:00 | 266.40 | 288.31 | 292.66 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 5 — BUY (started 2024-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-24 14:15:00 | 311.95 | 289.91 | 289.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-26 10:15:00 | 315.75 | 291.95 | 290.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 10:15:00 | 424.15 | 428.66 | 406.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-05 11:00:00 | 424.15 | 428.66 | 406.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 10:15:00 | 439.00 | 452.96 | 440.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 11:00:00 | 439.00 | 452.96 | 440.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 11:15:00 | 438.85 | 452.82 | 440.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 12:15:00 | 437.65 | 452.82 | 440.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 442.70 | 452.05 | 440.82 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2024-10-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 12:15:00 | 409.85 | 434.73 | 434.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 11:15:00 | 403.65 | 433.16 | 433.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 11:15:00 | 401.35 | 399.01 | 410.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-25 12:00:00 | 401.35 | 399.01 | 410.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 14:15:00 | 409.70 | 400.19 | 410.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 15:00:00 | 409.70 | 400.19 | 410.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 15:15:00 | 407.00 | 400.26 | 410.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 09:15:00 | 408.45 | 400.26 | 410.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 411.00 | 400.37 | 410.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:15:00 | 411.65 | 400.37 | 410.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 407.75 | 400.44 | 410.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 13:00:00 | 407.30 | 400.60 | 410.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 14:30:00 | 406.65 | 400.74 | 410.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-02 09:15:00 | 417.15 | 401.41 | 409.96 | SL hit (close>static) qty=1.00 sl=411.70 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 12:15:00 | 348.30 | 345.21 | 345.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 11:15:00 | 350.00 | 345.37 | 345.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 10:15:00 | 347.45 | 347.47 | 346.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 10:45:00 | 348.00 | 347.47 | 346.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 345.55 | 347.45 | 346.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:00:00 | 345.55 | 347.45 | 346.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 348.50 | 347.46 | 346.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:30:00 | 347.90 | 347.46 | 346.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 344.50 | 347.43 | 346.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 344.50 | 347.43 | 346.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 343.95 | 347.40 | 346.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 15:00:00 | 343.95 | 347.40 | 346.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 346.85 | 347.04 | 346.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 13:45:00 | 347.30 | 347.04 | 346.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 11:00:00 | 347.50 | 347.29 | 346.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 11:30:00 | 347.20 | 347.29 | 346.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 15:15:00 | 347.35 | 347.26 | 346.46 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 15:15:00 | 347.35 | 347.26 | 346.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:15:00 | 347.05 | 347.26 | 346.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 345.30 | 347.24 | 346.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:00:00 | 345.30 | 347.24 | 346.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 344.55 | 347.21 | 346.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:30:00 | 344.25 | 347.21 | 346.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-19 11:15:00 | 341.85 | 347.16 | 346.43 | SL hit (close<static) qty=1.00 sl=344.35 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 09:15:00 | 334.45 | 346.86 | 346.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 330.95 | 344.78 | 345.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 14:15:00 | 329.85 | 329.39 | 335.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-19 15:00:00 | 329.85 | 329.39 | 335.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 333.55 | 327.38 | 333.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 333.55 | 327.38 | 333.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 331.15 | 327.42 | 333.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 12:30:00 | 329.35 | 327.43 | 333.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 14:00:00 | 330.55 | 327.47 | 333.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 14:15:00 | 335.35 | 327.87 | 333.14 | SL hit (close>static) qty=1.00 sl=334.40 alert=retest2 |

### Cycle 9 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 277.85 | 250.24 | 250.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 10:15:00 | 278.80 | 250.53 | 250.37 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-07-26 09:15:00 | 288.15 | 2023-08-16 09:15:00 | 284.60 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2023-07-26 11:00:00 | 288.25 | 2023-08-16 09:15:00 | 284.60 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2023-07-26 11:30:00 | 288.35 | 2023-08-16 09:15:00 | 284.60 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2023-07-27 14:45:00 | 288.25 | 2023-08-16 09:15:00 | 284.60 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2023-08-03 11:30:00 | 295.00 | 2023-08-16 09:15:00 | 284.60 | STOP_HIT | 1.00 | -3.53% |
| BUY | retest2 | 2023-08-04 09:30:00 | 294.25 | 2023-08-16 09:15:00 | 284.60 | STOP_HIT | 1.00 | -3.28% |
| BUY | retest2 | 2023-08-14 11:15:00 | 297.05 | 2023-08-16 09:15:00 | 284.60 | STOP_HIT | 1.00 | -4.19% |
| BUY | retest2 | 2023-08-22 11:30:00 | 294.40 | 2023-10-09 09:15:00 | 297.80 | STOP_HIT | 1.00 | 1.15% |
| BUY | retest2 | 2023-09-28 09:15:00 | 304.40 | 2023-10-09 09:15:00 | 297.80 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2023-10-05 14:45:00 | 300.45 | 2023-10-09 09:15:00 | 297.80 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2023-10-06 10:15:00 | 300.65 | 2023-10-09 09:15:00 | 297.80 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2023-10-06 12:00:00 | 300.50 | 2023-10-19 09:15:00 | 295.50 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2023-10-11 09:15:00 | 303.40 | 2023-10-19 09:15:00 | 295.50 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2023-10-12 09:15:00 | 300.90 | 2023-10-19 09:15:00 | 295.50 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2023-10-13 14:45:00 | 300.05 | 2023-10-19 09:15:00 | 295.50 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2023-10-16 09:15:00 | 303.00 | 2023-10-20 11:15:00 | 286.20 | STOP_HIT | 1.00 | -5.54% |
| SELL | retest2 | 2023-11-22 12:00:00 | 289.75 | 2023-12-04 10:15:00 | 291.65 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2023-11-23 09:30:00 | 289.70 | 2023-12-04 10:15:00 | 291.65 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2023-12-01 15:15:00 | 289.00 | 2023-12-04 10:15:00 | 291.65 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2023-12-15 11:15:00 | 289.30 | 2023-12-18 09:15:00 | 294.15 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2023-12-18 11:30:00 | 293.00 | 2023-12-19 11:15:00 | 295.10 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2023-12-18 12:15:00 | 293.35 | 2023-12-19 11:15:00 | 295.10 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2023-12-18 14:30:00 | 293.30 | 2023-12-19 11:15:00 | 295.10 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2023-12-19 09:30:00 | 292.50 | 2023-12-19 11:15:00 | 295.10 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2024-01-24 11:45:00 | 306.30 | 2024-01-25 10:15:00 | 302.40 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2024-02-02 10:45:00 | 307.20 | 2024-02-05 14:15:00 | 302.15 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2024-02-02 11:30:00 | 307.00 | 2024-02-05 14:15:00 | 302.15 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2024-02-05 09:45:00 | 306.40 | 2024-02-05 14:15:00 | 302.15 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2024-03-05 12:15:00 | 296.00 | 2024-03-13 11:15:00 | 281.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-05 13:00:00 | 296.00 | 2024-03-13 11:15:00 | 281.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-05 13:30:00 | 295.25 | 2024-03-13 13:15:00 | 280.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-07 11:45:00 | 295.50 | 2024-03-13 13:15:00 | 280.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-05 12:15:00 | 296.00 | 2024-03-19 14:15:00 | 266.40 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-03-05 13:00:00 | 296.00 | 2024-03-19 14:15:00 | 266.40 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-03-05 13:30:00 | 295.25 | 2024-03-20 09:15:00 | 265.73 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-03-07 11:45:00 | 295.50 | 2024-03-20 09:15:00 | 265.95 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-04-09 14:00:00 | 286.65 | 2024-04-10 13:15:00 | 290.20 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2024-04-09 14:45:00 | 286.80 | 2024-04-10 13:15:00 | 290.20 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-04-10 09:45:00 | 286.90 | 2024-04-10 13:15:00 | 290.20 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2024-04-10 11:15:00 | 286.20 | 2024-04-10 13:15:00 | 290.20 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2024-11-28 13:00:00 | 407.30 | 2024-12-02 09:15:00 | 417.15 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2024-11-28 14:30:00 | 406.65 | 2024-12-02 09:15:00 | 417.15 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2024-12-03 09:30:00 | 407.35 | 2024-12-09 09:15:00 | 411.60 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2024-12-03 10:00:00 | 405.85 | 2024-12-09 09:15:00 | 411.60 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2024-12-04 10:45:00 | 408.40 | 2024-12-09 10:15:00 | 413.55 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2024-12-04 15:15:00 | 408.00 | 2024-12-09 10:15:00 | 413.55 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2024-12-12 12:00:00 | 407.55 | 2024-12-13 14:15:00 | 410.95 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2024-12-16 10:45:00 | 407.95 | 2024-12-20 14:15:00 | 387.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 10:45:00 | 407.95 | 2025-01-03 09:15:00 | 367.15 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-13 13:00:00 | 348.10 | 2025-03-18 14:15:00 | 357.10 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2025-04-01 11:15:00 | 348.55 | 2025-04-04 09:15:00 | 331.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-01 12:45:00 | 347.45 | 2025-04-04 09:15:00 | 330.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-01 11:15:00 | 348.55 | 2025-04-07 09:15:00 | 313.69 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-01 12:45:00 | 347.45 | 2025-04-07 09:15:00 | 312.70 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-23 09:15:00 | 344.50 | 2025-05-02 11:15:00 | 327.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-23 09:15:00 | 344.50 | 2025-05-16 09:15:00 | 347.35 | STOP_HIT | 0.50 | -0.83% |
| SELL | retest2 | 2025-05-21 09:45:00 | 344.15 | 2025-05-21 10:15:00 | 348.10 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-06-16 13:45:00 | 347.30 | 2025-06-19 11:15:00 | 341.85 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-06-18 11:00:00 | 347.50 | 2025-06-19 11:15:00 | 341.85 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-06-18 11:30:00 | 347.20 | 2025-06-19 11:15:00 | 341.85 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-06-18 15:15:00 | 347.35 | 2025-06-19 11:15:00 | 341.85 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-07-04 10:45:00 | 354.90 | 2025-07-08 13:15:00 | 343.60 | STOP_HIT | 1.00 | -3.18% |
| BUY | retest2 | 2025-07-04 15:00:00 | 355.05 | 2025-07-08 13:15:00 | 343.60 | STOP_HIT | 1.00 | -3.22% |
| SELL | retest2 | 2025-08-29 12:30:00 | 329.35 | 2025-09-01 14:15:00 | 335.35 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-08-29 14:00:00 | 330.55 | 2025-09-01 14:15:00 | 335.35 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-09-04 11:15:00 | 330.55 | 2025-09-12 14:15:00 | 314.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-04 13:15:00 | 330.50 | 2025-09-12 14:15:00 | 313.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-09 09:15:00 | 323.75 | 2025-09-22 14:15:00 | 307.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-04 11:15:00 | 330.55 | 2025-09-25 10:15:00 | 297.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-04 13:15:00 | 330.50 | 2025-09-25 10:15:00 | 297.45 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-09 09:15:00 | 323.75 | 2025-09-26 11:15:00 | 291.38 | TARGET_HIT | 0.50 | 10.00% |
