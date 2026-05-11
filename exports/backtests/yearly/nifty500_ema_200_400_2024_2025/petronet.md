# Petronet LNG Ltd. (PETRONET)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 282.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 2 |
| ALERT3 | 44 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 47 |
| PARTIAL | 8 |
| TARGET_HIT | 10 |
| STOP_HIT | 38 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 55 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 37
- **Target hits / Stop hits / Partials:** 9 / 38 / 8
- **Avg / median % per leg:** 1.04% / -0.85%
- **Sum % (uncompounded):** 57.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 2 | 11.1% | 2 | 16 | 0 | -0.37% | -6.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 18 | 2 | 11.1% | 2 | 16 | 0 | -0.37% | -6.7% |
| SELL (all) | 37 | 16 | 43.2% | 7 | 22 | 8 | 1.72% | 63.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 37 | 16 | 43.2% | 7 | 22 | 8 | 1.72% | 63.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 55 | 18 | 32.7% | 9 | 38 | 8 | 1.04% | 57.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 11:15:00 | 330.00 | 344.58 | 344.60 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-03 13:15:00 | 363.50 | 344.55 | 344.54 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 10:15:00 | 326.00 | 346.02 | 346.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 09:15:00 | 322.35 | 340.40 | 342.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 10:15:00 | 332.60 | 331.87 | 337.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-25 11:00:00 | 332.60 | 331.87 | 337.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 15:15:00 | 337.00 | 331.20 | 336.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:15:00 | 338.90 | 331.20 | 336.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 339.05 | 331.28 | 336.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 09:15:00 | 334.95 | 333.21 | 336.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 09:45:00 | 334.95 | 333.23 | 336.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 14:00:00 | 334.90 | 333.36 | 336.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 14:45:00 | 334.80 | 333.38 | 336.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 335.80 | 333.42 | 336.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 12:00:00 | 333.40 | 333.45 | 336.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 13:15:00 | 333.80 | 333.45 | 336.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-11 09:15:00 | 337.80 | 333.53 | 336.45 | SL hit (close>static) qty=1.00 sl=336.70 alert=retest2 |

### Cycle 4 — BUY (started 2025-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 10:15:00 | 345.45 | 337.99 | 337.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 11:15:00 | 346.55 | 338.08 | 338.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 09:15:00 | 326.70 | 338.35 | 338.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-02 09:15:00 | 326.70 | 338.35 | 338.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 326.70 | 338.35 | 338.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 10:00:00 | 326.70 | 338.35 | 338.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 10:15:00 | 321.95 | 338.19 | 338.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 11:00:00 | 321.95 | 338.19 | 338.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2025-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 12:15:00 | 323.10 | 337.90 | 337.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 13:15:00 | 321.85 | 334.89 | 336.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 12:15:00 | 329.95 | 329.86 | 332.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 12:30:00 | 329.50 | 329.86 | 332.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 13:15:00 | 304.05 | 293.75 | 303.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 14:00:00 | 304.05 | 293.75 | 303.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 14:15:00 | 301.45 | 293.83 | 303.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 10:30:00 | 299.85 | 294.73 | 303.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 10:45:00 | 299.25 | 294.98 | 303.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 12:15:00 | 284.86 | 294.86 | 301.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 12:15:00 | 284.29 | 294.86 | 301.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-04-07 09:15:00 | 269.87 | 294.33 | 301.09 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 6 — BUY (started 2025-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 15:15:00 | 315.00 | 302.87 | 302.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 315.40 | 304.69 | 303.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 15:15:00 | 312.30 | 312.34 | 308.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-29 09:15:00 | 311.30 | 312.34 | 308.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 308.80 | 312.31 | 308.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:45:00 | 309.40 | 312.31 | 308.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 308.75 | 312.28 | 308.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:00:00 | 308.75 | 312.28 | 308.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 12:15:00 | 308.05 | 312.24 | 308.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 13:00:00 | 308.05 | 312.24 | 308.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 13:15:00 | 308.25 | 312.20 | 308.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 09:30:00 | 309.20 | 310.90 | 308.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 09:15:00 | 306.80 | 310.70 | 308.54 | SL hit (close<static) qty=1.00 sl=307.90 alert=retest2 |

### Cycle 7 — SELL (started 2025-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 09:15:00 | 295.30 | 307.16 | 307.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 12:15:00 | 292.00 | 306.78 | 306.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-27 09:15:00 | 305.00 | 303.70 | 305.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 09:15:00 | 305.00 | 303.70 | 305.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 305.00 | 303.70 | 305.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 12:45:00 | 301.00 | 303.60 | 305.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 11:45:00 | 300.95 | 303.51 | 305.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 13:15:00 | 299.30 | 303.49 | 305.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 11:30:00 | 300.90 | 303.27 | 304.86 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 302.55 | 302.75 | 304.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:15:00 | 305.70 | 302.75 | 304.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 307.40 | 302.80 | 304.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:00:00 | 307.40 | 302.80 | 304.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 306.70 | 302.84 | 304.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:30:00 | 307.25 | 302.84 | 304.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 304.25 | 303.05 | 304.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:30:00 | 305.00 | 303.05 | 304.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 304.70 | 303.07 | 304.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:30:00 | 305.30 | 303.07 | 304.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 303.80 | 303.07 | 304.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 10:15:00 | 303.25 | 303.26 | 304.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 11:45:00 | 303.55 | 303.27 | 304.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 13:15:00 | 305.25 | 303.31 | 304.54 | SL hit (close>static) qty=1.00 sl=305.05 alert=retest2 |

### Cycle 8 — BUY (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 09:15:00 | 288.35 | 277.23 | 277.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 11:15:00 | 293.30 | 278.24 | 277.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 09:15:00 | 282.00 | 282.29 | 280.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-19 10:00:00 | 282.00 | 282.29 | 280.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 281.25 | 282.28 | 280.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:30:00 | 281.00 | 282.28 | 280.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 11:15:00 | 279.65 | 282.25 | 280.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 11:30:00 | 279.85 | 282.25 | 280.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 12:15:00 | 278.60 | 282.22 | 280.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 13:00:00 | 278.60 | 282.22 | 280.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 279.05 | 282.12 | 280.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:15:00 | 279.40 | 282.12 | 280.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 277.15 | 282.02 | 280.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-23 09:30:00 | 281.35 | 280.75 | 279.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-23 13:00:00 | 279.05 | 280.70 | 279.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-23 14:30:00 | 279.00 | 280.65 | 279.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-23 15:15:00 | 279.15 | 280.65 | 279.63 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 279.15 | 280.63 | 279.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:15:00 | 276.10 | 280.63 | 279.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-27 12:15:00 | 276.40 | 280.51 | 279.58 | SL hit (close<static) qty=1.00 sl=276.95 alert=retest2 |

### Cycle 9 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 237.00 | 288.95 | 288.96 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-04 12:30:00 | 269.25 | 2024-06-06 09:15:00 | 296.18 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-09 09:15:00 | 334.95 | 2024-12-11 09:15:00 | 337.80 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2024-12-09 09:45:00 | 334.95 | 2024-12-11 09:15:00 | 337.80 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2024-12-09 14:00:00 | 334.90 | 2024-12-12 09:15:00 | 346.20 | STOP_HIT | 1.00 | -3.37% |
| SELL | retest2 | 2024-12-09 14:45:00 | 334.80 | 2024-12-12 09:15:00 | 346.20 | STOP_HIT | 1.00 | -3.41% |
| SELL | retest2 | 2024-12-10 12:00:00 | 333.40 | 2024-12-12 09:15:00 | 346.20 | STOP_HIT | 1.00 | -3.84% |
| SELL | retest2 | 2024-12-10 13:15:00 | 333.80 | 2024-12-12 09:15:00 | 346.20 | STOP_HIT | 1.00 | -3.71% |
| SELL | retest2 | 2024-12-18 11:30:00 | 333.65 | 2024-12-19 11:15:00 | 337.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2024-12-19 09:15:00 | 330.50 | 2024-12-19 11:15:00 | 337.00 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-03-25 10:30:00 | 299.85 | 2025-04-04 12:15:00 | 284.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-26 10:45:00 | 299.25 | 2025-04-04 12:15:00 | 284.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-25 10:30:00 | 299.85 | 2025-04-07 09:15:00 | 269.87 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-26 10:45:00 | 299.25 | 2025-04-15 09:15:00 | 294.20 | STOP_HIT | 0.50 | 1.69% |
| SELL | retest2 | 2025-04-16 14:15:00 | 299.80 | 2025-04-21 09:15:00 | 310.05 | STOP_HIT | 1.00 | -3.42% |
| SELL | retest2 | 2025-04-17 10:00:00 | 299.55 | 2025-04-21 09:15:00 | 310.05 | STOP_HIT | 1.00 | -3.51% |
| BUY | retest2 | 2025-06-05 09:30:00 | 309.20 | 2025-06-06 09:15:00 | 306.80 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-06-09 11:45:00 | 309.10 | 2025-06-12 09:15:00 | 307.80 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-06-10 10:00:00 | 309.00 | 2025-06-12 09:15:00 | 307.80 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-06-11 09:15:00 | 309.85 | 2025-06-12 09:15:00 | 307.80 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-06-30 12:45:00 | 301.00 | 2025-07-10 13:15:00 | 305.25 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-07-01 11:45:00 | 300.95 | 2025-07-10 13:15:00 | 305.25 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-07-01 13:15:00 | 299.30 | 2025-07-14 11:15:00 | 305.60 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2025-07-02 11:30:00 | 300.90 | 2025-07-15 09:15:00 | 307.15 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-07-10 10:15:00 | 303.25 | 2025-07-15 13:15:00 | 311.10 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2025-07-10 11:45:00 | 303.55 | 2025-07-15 13:15:00 | 311.10 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2025-07-11 10:00:00 | 302.40 | 2025-07-15 13:15:00 | 311.10 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2025-07-14 13:30:00 | 303.50 | 2025-07-15 13:15:00 | 311.10 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2025-07-18 12:45:00 | 304.90 | 2025-07-28 09:15:00 | 306.25 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-07-23 13:00:00 | 305.05 | 2025-07-28 09:15:00 | 306.25 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-07-25 10:15:00 | 304.85 | 2025-07-28 09:15:00 | 306.25 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-07-25 11:15:00 | 304.45 | 2025-07-31 09:15:00 | 289.65 | PARTIAL | 0.50 | 4.86% |
| SELL | retest2 | 2025-07-25 12:15:00 | 303.20 | 2025-07-31 09:15:00 | 289.80 | PARTIAL | 0.50 | 4.42% |
| SELL | retest2 | 2025-07-25 13:00:00 | 303.25 | 2025-07-31 09:15:00 | 289.61 | PARTIAL | 0.50 | 4.50% |
| SELL | retest2 | 2025-07-25 14:45:00 | 303.00 | 2025-07-31 09:15:00 | 289.23 | PARTIAL | 0.50 | 4.55% |
| SELL | retest2 | 2025-07-28 10:45:00 | 303.20 | 2025-07-31 09:15:00 | 288.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-28 12:45:00 | 303.00 | 2025-07-31 09:15:00 | 287.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-25 11:15:00 | 304.45 | 2025-08-06 10:15:00 | 274.41 | TARGET_HIT | 0.50 | 9.87% |
| SELL | retest2 | 2025-07-25 12:15:00 | 303.20 | 2025-08-06 10:15:00 | 274.55 | TARGET_HIT | 0.50 | 9.45% |
| SELL | retest2 | 2025-07-25 13:00:00 | 303.25 | 2025-08-06 10:15:00 | 274.37 | TARGET_HIT | 0.50 | 9.53% |
| SELL | retest2 | 2025-07-25 14:45:00 | 303.00 | 2025-08-06 10:15:00 | 274.00 | TARGET_HIT | 0.50 | 9.57% |
| SELL | retest2 | 2025-07-28 10:45:00 | 303.20 | 2025-08-06 11:15:00 | 272.88 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-28 12:45:00 | 303.00 | 2025-08-06 11:15:00 | 272.70 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-01-23 09:30:00 | 281.35 | 2026-01-27 12:15:00 | 276.40 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2026-01-23 13:00:00 | 279.05 | 2026-01-27 12:15:00 | 276.40 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2026-01-23 14:30:00 | 279.00 | 2026-01-27 12:15:00 | 276.40 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2026-01-23 15:15:00 | 279.15 | 2026-01-27 12:15:00 | 276.40 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2026-02-02 09:15:00 | 285.05 | 2026-02-25 10:15:00 | 313.56 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-04 13:15:00 | 281.95 | 2026-03-04 14:15:00 | 279.75 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2026-03-04 13:45:00 | 281.90 | 2026-03-04 14:15:00 | 279.75 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2026-03-04 14:45:00 | 282.00 | 2026-03-04 15:15:00 | 279.20 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2026-03-12 10:15:00 | 297.50 | 2026-03-13 10:15:00 | 287.95 | STOP_HIT | 1.00 | -3.21% |
| BUY | retest2 | 2026-03-12 11:30:00 | 296.90 | 2026-03-13 10:15:00 | 287.95 | STOP_HIT | 1.00 | -3.01% |
| BUY | retest2 | 2026-03-12 13:30:00 | 297.70 | 2026-03-13 10:15:00 | 287.95 | STOP_HIT | 1.00 | -3.28% |
| BUY | retest2 | 2026-03-12 14:45:00 | 296.75 | 2026-03-13 10:15:00 | 287.95 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest2 | 2026-03-18 12:15:00 | 291.90 | 2026-03-19 09:15:00 | 277.75 | STOP_HIT | 1.00 | -4.85% |
