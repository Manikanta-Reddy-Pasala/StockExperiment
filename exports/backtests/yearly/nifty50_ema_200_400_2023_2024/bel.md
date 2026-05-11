# BEL (BEL)

## Backtest Summary

- **Window:** 2022-04-07 09:15:00 → 2026-05-08 15:15:00 (7054 bars)
- **Last close:** 439.50
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
| ALERT2 | 9 |
| ALERT2_SKIP | 2 |
| ALERT3 | 45 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 56 |
| PARTIAL | 10 |
| TARGET_HIT | 6 |
| STOP_HIT | 47 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 63 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 23 / 40
- **Target hits / Stop hits / Partials:** 6 / 47 / 10
- **Avg / median % per leg:** -0.04% / -1.08%
- **Sum % (uncompounded):** -2.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 33 | 4 | 12.1% | 4 | 29 | 0 | -1.31% | -43.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 33 | 4 | 12.1% | 4 | 29 | 0 | -1.31% | -43.3% |
| SELL (all) | 30 | 19 | 63.3% | 2 | 18 | 10 | 1.36% | 40.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 30 | 19 | 63.3% | 2 | 18 | 10 | 1.36% | 40.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 63 | 23 | 36.5% | 6 | 47 | 10 | -0.04% | -2.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 15:15:00 | 282.60 | 294.52 | 294.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 09:15:00 | 275.05 | 294.33 | 294.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-24 13:15:00 | 292.15 | 291.30 | 292.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-24 13:45:00 | 292.05 | 291.30 | 292.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 15:15:00 | 292.55 | 291.32 | 292.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 09:15:00 | 293.05 | 291.32 | 292.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 289.50 | 291.30 | 292.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-25 11:45:00 | 289.25 | 291.26 | 292.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-25 13:00:00 | 289.15 | 291.24 | 292.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-25 14:15:00 | 289.10 | 291.23 | 292.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 10:45:00 | 289.25 | 291.19 | 292.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 291.85 | 291.14 | 292.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 11:00:00 | 291.15 | 291.14 | 292.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-27 14:15:00 | 293.75 | 291.16 | 292.59 | SL hit (close>static) qty=1.00 sl=293.55 alert=retest2 |

### Cycle 2 — BUY (started 2024-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 09:15:00 | 308.60 | 287.77 | 287.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 11:15:00 | 309.95 | 291.73 | 289.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 11:15:00 | 303.05 | 303.44 | 297.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-18 12:00:00 | 303.05 | 303.44 | 297.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 297.65 | 303.38 | 297.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 11:15:00 | 299.90 | 303.33 | 297.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 09:30:00 | 301.25 | 303.08 | 297.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 10:30:00 | 299.70 | 303.05 | 297.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-20 13:15:00 | 293.30 | 302.86 | 297.56 | SL hit (close<static) qty=1.00 sl=295.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-01-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 15:15:00 | 281.45 | 294.53 | 294.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 09:15:00 | 273.85 | 294.32 | 294.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 12:15:00 | 286.90 | 286.88 | 290.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-20 13:00:00 | 286.90 | 286.88 | 290.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 288.20 | 279.51 | 285.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 10:00:00 | 288.20 | 279.51 | 285.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 10:15:00 | 291.15 | 279.62 | 285.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 11:00:00 | 291.15 | 279.62 | 285.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 273.35 | 280.61 | 285.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 09:30:00 | 265.00 | 280.50 | 285.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 09:15:00 | 269.00 | 280.23 | 284.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 10:00:00 | 268.05 | 280.11 | 284.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 255.55 | 279.02 | 283.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 10:15:00 | 254.65 | 278.80 | 283.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 11:15:00 | 251.75 | 276.22 | 281.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-18 12:15:00 | 242.10 | 272.10 | 279.19 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2025-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 15:15:00 | 303.10 | 276.72 | 276.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-26 11:15:00 | 304.61 | 279.04 | 277.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-02 09:15:00 | 275.50 | 283.61 | 280.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 09:15:00 | 275.50 | 283.61 | 280.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 275.50 | 283.61 | 280.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-02 10:00:00 | 275.50 | 283.61 | 280.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 280.15 | 283.58 | 280.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 13:30:00 | 281.55 | 283.50 | 280.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 14:15:00 | 282.05 | 283.50 | 280.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 10:30:00 | 283.30 | 283.66 | 280.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 266.90 | 283.36 | 280.65 | SL hit (close<static) qty=1.00 sl=274.50 alert=retest2 |

### Cycle 5 — SELL (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 09:15:00 | 375.20 | 382.29 | 382.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 13:15:00 | 372.70 | 381.44 | 381.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 10:15:00 | 381.05 | 379.46 | 380.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 10:15:00 | 381.05 | 379.46 | 380.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 381.05 | 379.46 | 380.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:45:00 | 380.50 | 379.46 | 380.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 384.20 | 379.51 | 380.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:00:00 | 384.20 | 379.51 | 380.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 386.10 | 379.57 | 380.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 13:00:00 | 386.10 | 379.57 | 380.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2025-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 11:15:00 | 397.65 | 381.90 | 381.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 400.75 | 382.69 | 382.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 14:15:00 | 407.00 | 407.56 | 400.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-29 15:00:00 | 407.00 | 407.56 | 400.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 408.05 | 416.02 | 408.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:00:00 | 408.05 | 416.02 | 408.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 407.45 | 415.93 | 408.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:45:00 | 407.80 | 415.93 | 408.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 403.85 | 415.65 | 408.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 403.85 | 415.65 | 408.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 409.80 | 415.39 | 408.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 12:00:00 | 410.95 | 415.35 | 408.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 09:15:00 | 411.15 | 415.17 | 408.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 15:00:00 | 411.55 | 414.84 | 409.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 407.75 | 414.69 | 409.45 | SL hit (close<static) qty=1.00 sl=408.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 11:15:00 | 390.90 | 405.68 | 405.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 13:15:00 | 389.90 | 405.37 | 405.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 09:15:00 | 400.50 | 400.36 | 402.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:45:00 | 401.65 | 400.36 | 402.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 402.25 | 400.41 | 402.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:30:00 | 402.20 | 400.41 | 402.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 12:15:00 | 401.85 | 400.43 | 402.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 12:30:00 | 402.60 | 400.43 | 402.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 404.35 | 400.46 | 402.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 12:30:00 | 402.10 | 400.54 | 402.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 11:45:00 | 402.25 | 399.57 | 401.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 415.65 | 399.85 | 401.89 | SL hit (close>static) qty=1.00 sl=407.55 alert=retest2 |

### Cycle 8 — BUY (started 2026-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 10:15:00 | 420.05 | 403.77 | 403.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 10:15:00 | 430.40 | 408.91 | 406.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 440.90 | 444.04 | 433.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-13 10:00:00 | 440.90 | 444.04 | 433.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 432.80 | 443.62 | 433.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-17 13:30:00 | 439.40 | 442.24 | 433.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 10:30:00 | 438.60 | 442.12 | 433.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 410.20 | 440.55 | 433.36 | SL hit (close<static) qty=1.00 sl=425.70 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-10-20 14:15:00 | 134.55 | 2023-10-23 09:15:00 | 131.80 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2023-10-23 09:15:00 | 135.60 | 2023-10-23 09:15:00 | 131.80 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2023-11-02 11:15:00 | 134.65 | 2023-11-02 11:15:00 | 133.80 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2023-11-02 13:45:00 | 134.55 | 2023-12-01 09:15:00 | 148.01 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-03-14 10:30:00 | 193.75 | 2024-03-15 10:15:00 | 184.10 | STOP_HIT | 1.00 | -4.98% |
| BUY | retest2 | 2024-03-14 12:15:00 | 193.65 | 2024-03-15 10:15:00 | 184.10 | STOP_HIT | 1.00 | -4.93% |
| BUY | retest2 | 2024-03-14 13:00:00 | 193.40 | 2024-03-15 10:15:00 | 184.10 | STOP_HIT | 1.00 | -4.81% |
| BUY | retest2 | 2024-03-14 15:00:00 | 194.80 | 2024-03-15 10:15:00 | 184.10 | STOP_HIT | 1.00 | -5.49% |
| BUY | retest2 | 2024-03-18 09:15:00 | 190.50 | 2024-03-19 09:15:00 | 188.10 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2024-03-18 12:00:00 | 190.00 | 2024-03-19 09:15:00 | 188.10 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2024-03-18 13:00:00 | 190.15 | 2024-03-19 09:15:00 | 188.10 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2024-03-18 14:15:00 | 191.15 | 2024-03-19 09:15:00 | 188.10 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2024-06-06 09:15:00 | 276.30 | 2024-06-14 09:15:00 | 303.93 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-09-25 11:45:00 | 289.25 | 2024-09-27 14:15:00 | 293.75 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2024-09-25 13:00:00 | 289.15 | 2024-10-03 09:15:00 | 276.31 | PARTIAL | 0.50 | 4.44% |
| SELL | retest2 | 2024-09-25 14:15:00 | 289.10 | 2024-10-04 09:15:00 | 274.79 | PARTIAL | 0.50 | 4.95% |
| SELL | retest2 | 2024-09-26 10:45:00 | 289.25 | 2024-10-04 09:15:00 | 274.69 | PARTIAL | 0.50 | 5.03% |
| SELL | retest2 | 2024-09-27 11:00:00 | 291.15 | 2024-10-04 09:15:00 | 274.64 | PARTIAL | 0.50 | 5.67% |
| SELL | retest2 | 2024-09-27 14:45:00 | 290.85 | 2024-10-04 09:15:00 | 274.79 | PARTIAL | 0.50 | 5.52% |
| SELL | retest2 | 2024-09-30 09:15:00 | 287.90 | 2024-10-07 09:15:00 | 273.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-25 13:00:00 | 289.15 | 2024-10-10 10:15:00 | 287.40 | STOP_HIT | 0.50 | 0.61% |
| SELL | retest2 | 2024-09-25 14:15:00 | 289.10 | 2024-10-10 10:15:00 | 287.40 | STOP_HIT | 0.50 | 0.59% |
| SELL | retest2 | 2024-09-26 10:45:00 | 289.25 | 2024-10-10 10:15:00 | 287.40 | STOP_HIT | 0.50 | 0.64% |
| SELL | retest2 | 2024-09-27 11:00:00 | 291.15 | 2024-10-10 10:15:00 | 287.40 | STOP_HIT | 0.50 | 1.29% |
| SELL | retest2 | 2024-09-27 14:45:00 | 290.85 | 2024-10-10 10:15:00 | 287.40 | STOP_HIT | 0.50 | 1.19% |
| SELL | retest2 | 2024-09-30 09:15:00 | 287.90 | 2024-10-10 10:15:00 | 287.40 | STOP_HIT | 0.50 | 0.17% |
| SELL | retest2 | 2024-10-30 09:30:00 | 290.85 | 2024-11-06 10:15:00 | 295.65 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2024-10-30 12:00:00 | 289.80 | 2024-11-06 10:15:00 | 295.65 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2024-10-30 13:30:00 | 289.95 | 2024-11-06 10:15:00 | 295.65 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2024-11-06 10:15:00 | 290.50 | 2024-11-06 10:15:00 | 295.65 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2024-11-12 15:00:00 | 289.95 | 2024-11-21 09:15:00 | 275.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 15:00:00 | 289.95 | 2024-11-25 09:15:00 | 294.75 | STOP_HIT | 0.50 | -1.66% |
| SELL | retest2 | 2024-11-14 09:15:00 | 278.75 | 2024-11-25 09:15:00 | 294.75 | STOP_HIT | 1.00 | -5.74% |
| SELL | retest2 | 2024-11-18 14:15:00 | 278.45 | 2024-11-25 09:15:00 | 294.75 | STOP_HIT | 1.00 | -5.85% |
| SELL | retest2 | 2024-11-19 15:00:00 | 278.55 | 2024-11-25 09:15:00 | 294.75 | STOP_HIT | 1.00 | -5.82% |
| BUY | retest2 | 2024-12-19 11:15:00 | 299.90 | 2024-12-20 13:15:00 | 293.30 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2024-12-20 09:30:00 | 301.25 | 2024-12-20 13:15:00 | 293.30 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2024-12-20 10:30:00 | 299.70 | 2024-12-20 13:15:00 | 293.30 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-02-03 09:30:00 | 265.00 | 2025-02-12 09:15:00 | 255.55 | PARTIAL | 0.50 | 3.57% |
| SELL | retest2 | 2025-02-11 09:15:00 | 269.00 | 2025-02-12 10:15:00 | 254.65 | PARTIAL | 0.50 | 5.34% |
| SELL | retest2 | 2025-02-11 10:00:00 | 268.05 | 2025-02-14 11:15:00 | 251.75 | PARTIAL | 0.50 | 6.08% |
| SELL | retest2 | 2025-02-03 09:30:00 | 265.00 | 2025-02-18 12:15:00 | 242.10 | TARGET_HIT | 0.50 | 8.64% |
| SELL | retest2 | 2025-02-11 09:15:00 | 269.00 | 2025-02-18 12:15:00 | 241.25 | TARGET_HIT | 0.50 | 10.32% |
| SELL | retest2 | 2025-02-11 10:00:00 | 268.05 | 2025-03-04 10:15:00 | 264.20 | STOP_HIT | 0.50 | 1.44% |
| BUY | retest2 | 2025-04-02 13:30:00 | 281.55 | 2025-04-07 09:15:00 | 266.90 | STOP_HIT | 1.00 | -5.20% |
| BUY | retest2 | 2025-04-02 14:15:00 | 282.05 | 2025-04-07 09:15:00 | 266.90 | STOP_HIT | 1.00 | -5.37% |
| BUY | retest2 | 2025-04-04 10:30:00 | 283.30 | 2025-04-07 09:15:00 | 266.90 | STOP_HIT | 1.00 | -5.79% |
| BUY | retest2 | 2025-04-08 09:15:00 | 285.55 | 2025-04-23 09:15:00 | 307.89 | TARGET_HIT | 1.00 | 7.82% |
| BUY | retest2 | 2025-04-09 11:15:00 | 279.90 | 2025-04-29 09:15:00 | 314.11 | TARGET_HIT | 1.00 | 12.22% |
| BUY | retest2 | 2025-11-25 12:00:00 | 410.95 | 2025-12-03 09:15:00 | 407.75 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-11-26 09:15:00 | 411.15 | 2025-12-03 09:15:00 | 407.75 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-11-28 15:00:00 | 411.55 | 2025-12-03 09:15:00 | 407.75 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-12-26 12:30:00 | 402.10 | 2026-01-05 09:15:00 | 415.65 | STOP_HIT | 1.00 | -3.37% |
| SELL | retest2 | 2026-01-02 11:45:00 | 402.25 | 2026-01-05 09:15:00 | 415.65 | STOP_HIT | 1.00 | -3.33% |
| BUY | retest2 | 2026-03-17 13:30:00 | 439.40 | 2026-03-23 09:15:00 | 410.20 | STOP_HIT | 1.00 | -6.65% |
| BUY | retest2 | 2026-03-19 10:30:00 | 438.60 | 2026-03-23 09:15:00 | 410.20 | STOP_HIT | 1.00 | -6.48% |
| BUY | retest2 | 2026-04-09 09:15:00 | 440.40 | 2026-04-30 09:15:00 | 426.60 | STOP_HIT | 1.00 | -3.13% |
| BUY | retest2 | 2026-04-09 10:00:00 | 438.40 | 2026-04-30 09:15:00 | 426.60 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2026-04-28 09:15:00 | 438.95 | 2026-04-30 09:15:00 | 426.60 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2026-04-28 11:30:00 | 438.05 | 2026-04-30 09:15:00 | 426.60 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2026-04-28 14:30:00 | 437.05 | 2026-05-07 09:15:00 | 434.25 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2026-04-29 09:15:00 | 438.55 | 2026-05-07 09:15:00 | 434.25 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2026-05-06 09:30:00 | 438.10 | 2026-05-07 09:15:00 | 434.25 | STOP_HIT | 1.00 | -0.88% |
